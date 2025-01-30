from datetime import datetime
from typing import Optional
def calculate_age_group(birthday: Optional[datetime]) -> str:
    if birthday is None:
        return "18_29"

    today = datetime.today()
    age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))

    if age < 18:
        return "18+"
    elif 18 <= age < 30:
        return "18_29"
    elif 30 <= age < 45:
        return "30_44"
    elif 45 <= age < 60:
        return "45_59"
    else:
        return "60+"