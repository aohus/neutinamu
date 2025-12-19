import io
import random
import time
from typing import List

from locust import HttpUser, task, between, events
from PIL import Image

class ClusteringUser(HttpUser):
    # Wait time between tasks (simulating user thinking time)
    wait_time = between(2, 5)

    def on_start(self):
        """
        Generate a dummy image once per user to reuse.
        """
        self.dummy_image_data = self._generate_dummy_image()
        self.headers = {}
        # If authentication is needed:
        # response = self.client.post("/api/login", json={"username": "test", "password": "password"})
        # self.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

    def _generate_dummy_image(self) -> bytes:
        """Creates a small 100x100 random color JPEG image in memory."""
        img = Image.new('RGB', (100, 100), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return buf.getvalue()

    @task
    def run_clustering_scenario(self):
        """
        Full flow: Create Job -> Upload 200 Photos -> Start Clustering -> Wait for Completion
        """
        job_id = self.create_job()
        if not job_id:
            return

        success = self.upload_photos(job_id, count=200)
        if not success:
            return

        self.start_clustering(job_id)
        
        # Optional: Poll for completion to measure full processing time
        # self.poll_completion(job_id)

    def create_job(self):
        payload = {
            "title": f"PerfTest_User_{self.environment.runner.user_count}_{random.randint(1000, 9999)}",
            "construction_type": "Road Construction",
            "company_name": "Performance Corp"
        }
        with self.client.post("/api/jobs", json=payload, headers=self.headers, catch_response=True) as response:
            if response.status_code == 201:
                return response.json()["id"]
            else:
                response.failure(f"Failed to create job: {response.text}")
                return None

    def upload_photos(self, job_id, count=200):
        # Upload in batches to avoid overwhelming the connection/server in one go
        batch_size = 50
        uploaded = 0
        
        for i in range(0, count, batch_size):
            files = []
            current_batch = min(batch_size, count - uploaded)
            
            for j in range(current_batch):
                filename = f"photo_{uploaded + j}.jpg"
                # ('files', (filename, file_bytes, content_type))
                files.append(('files', (filename, self.dummy_image_data, 'image/jpeg')))
            
            with self.client.post(f"/api/jobs/{job_id}/photos", files=files, headers=self.headers, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Failed to upload batch {i}: {response.text}")
                    return False
                uploaded += current_batch
                
        return True

    def start_clustering(self, job_id):
        payload = {
            "min_samples": 3,
            "max_dist_m": 15.0,
            "max_alt_diff_m": 10.0
        }
        with self.client.post(f"/api/jobs/{job_id}/cluster", json=payload, headers=self.headers, catch_response=True) as response:
            if response.status_code == 202:
                # 202 Accepted is the expected response for starting a background task
                pass 
            else:
                response.failure(f"Failed to start clustering: {response.text}")

    def poll_completion(self, job_id):
        start_time = time.time()
        # Timeout after 5 minutes
        while time.time() - start_time < 300:
            response = self.client.get(f"/api/jobs/{job_id}", headers=self.headers)
            if response.status_code == 200:
                status = response.json()["status"]
                if status == "COMPLETED":
                    events.request.fire(
                        request_type="FLOW",
                        name="Clustering_Complete",
                        response_time=(time.time() - start_time) * 1000,
                        response_length=0,
                    )
                    return
                elif status == "FAILED":
                    events.request.fire(
                        request_type="FLOW",
                        name="Clustering_Failed",
                        response_time=(time.time() - start_time) * 1000,
                        exception=Exception("Job failed"),
                    )
                    return
            time.sleep(2)
