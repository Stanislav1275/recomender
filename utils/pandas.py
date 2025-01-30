from typing import Any, ClassVar

from fastapi.openapi.models import Response


class DataFrameJSONResponse(Response):
    media_type : ClassVar[str] = "application/json"
    @staticmethod
    def render(self, content: Any) -> bytes:
        return content.to_json(orient="records", date_format='iso').encode("utf-8")