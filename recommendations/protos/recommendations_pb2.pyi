from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class UserRequest(_message.Message):
    __slots__ = ("user_id",)
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    user_id: int
    def __init__(self, user_id: _Optional[int] = ...) -> None: ...

class TitleRequest(_message.Message):
    __slots__ = ("title_id",)
    TITLE_ID_FIELD_NUMBER: _ClassVar[int]
    title_id: int
    def __init__(self, title_id: _Optional[int] = ...) -> None: ...

class RecommendationResponse(_message.Message):
    __slots__ = ("item_ids",)
    ITEM_IDS_FIELD_NUMBER: _ClassVar[int]
    item_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, item_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class TrainRequest(_message.Message):
    __slots__ = ("force_retrain",)
    FORCE_RETRAIN_FIELD_NUMBER: _ClassVar[int]
    force_retrain: bool
    def __init__(self, force_retrain: bool = ...) -> None: ...

class TrainResponse(_message.Message):
    __slots__ = ("success", "message", "version")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    version: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...
