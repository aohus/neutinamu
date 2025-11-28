import secrets
import string

def generate_short_id(prefix: str, length: int = 10) -> str:
    chars = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(chars) for _ in range(length))
    return f"{prefix}_{random_str}"