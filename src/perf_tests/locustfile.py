from locust import HttpUser, task, between
import random

class ReportUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """
        Simulate login or setting up headers.
        Here we assume a simple auth token or header if needed.
        For now, we'll just interact with public endpoints or mock auth.
        If your API requires a Bearer token, you would fetch it here.
        """
        # self.client.headers.update({"Authorization": "Bearer <token>"})
        pass

    @task(3)
    def get_jobs(self):
        self.client.get("/api/jobs")

    @task(1)
    def create_job(self):
        payload = {
            "title": f"Load Test Job {random.randint(1, 1000)}",
            "construction_type": "Load Testing",
            "company_name": "Locust Inc"
        }
        self.client.post("/api/jobs", json=payload)

    @task(1)
    def health_check(self):
        self.client.get("/health")
