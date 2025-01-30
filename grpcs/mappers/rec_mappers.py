from datetime import datetime

from grpcs.protos import rec_pb2


class UserMapper:
    @staticmethod
    def to_grpc(user):
        return rec_pb2.UserResponse(
            id=user.id,
            last_login=user.last_login.isoformat() if user.last_login else "",
            is_superuser=user.is_superuser,
            is_staff=user.is_staff,
            is_active=user.is_active,
            date_joined=user.date_joined.isoformat() if user.date_joined else "",
            last_seen_date=user.last_seen_date.isoformat() if user.last_seen_date else "",
            yaoi=user.yaoi,
            adult=user.adult,
            preference=user.preference,
            is_banned=user.is_banned,
            birthday=user.birthday.isoformat() if user.birthday else "",
            sex=user.sex,
            is_premium=user.is_premium
        )