import random
import string

otp_cache = {}

def generate_otp():
    return "".join(random.choices(string.digits, k=6))
