import json
import logging
from pathlib import Path

import aiofiles

logger = logging.getLogger(__name__)


async def read_file(filepath: str) -> dict:
    file_path_obj = Path(filepath)
    if not file_path_obj.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

    suffix = file_path_obj.suffix.lower()
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        content = await f.read()
        if not content:
            return {} if suffix == ".json" else ""
        if suffix == ".json":
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 디코딩 오류: {e} - 파일: {filepath}") from e
            return data
        else:
            return content


async def write_file(filepath: str, data) -> None:
    file_path_obj = Path(filepath)
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)

    suffix = file_path_obj.suffix.lower()
    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
        if suffix == ".json":
            await f.write(json.dumps(data, ensure_ascii=False, indent=4))
        else:
            await f.write(str(data))
    logger.info(f"Write file successfully! File: {filepath}")

