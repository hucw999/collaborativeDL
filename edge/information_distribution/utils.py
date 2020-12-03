import random
import string


def generate_random_str(length: int) -> str:
    str_list = [random.choice(string.digits + string.ascii_letters) for _ in range(length)]
    random_str = ''.join(str_list)
    return random_str
