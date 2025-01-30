import json
import os
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import grpc
import threadpoolctl
import uvicorn
from google.protobuf.json_format import MessageToDict
from rectools import Columns

os.environ["OPENBLAS_NUM_THREADS"] = "1"
# warnings.filterwarnings('ignore')
threadpoolctl.threadpool_limits(1, "blas")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GRPC_VERBOSITY"] = "debug"
threadpoolctl.threadpool_limits(1)
from grpcs.protos import rec_pb2_grpc
from grpcs.protos.rec_pb2 import UserRetrieveRequest
from fastapi import Depends, FastAPI
from grpcs.services.rec import RecService
from grpcs.services.data_preparer import DataPrepareService

from core.database import SessionLocal
from fastapi.middleware.cors import CORSMiddleware
from grpcs.services.grpc_server import serve
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://0.0.0.0:50051"], allow_methods=["get", "post", "options"], allow_headers=["*"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root(user_id:int):
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = rec_pb2_grpc.RecServiceStub(channel)
        response = await stub.GetUser(UserRetrieveRequest(id=user_id))
        return {"data":MessageToDict(response,preserving_proto_field_name=True) }

@app.get("/users-features")
async def users_features():
    uf_pd=  await DataPrepareService.get_users_features()
    return uf_pd.to_html()
@app.get("/train")
async def train():
    with ThreadPoolExecutor() as executor:
        fut =executor.submit(RecService.train)
        await fut.result()
@app.get("/rec")
async def rec(user_id:int):
    with ThreadPoolExecutor() as executor:

        fut =executor.submit(RecService.rec, user_id)
        res = await fut.result()
        item_ids = res[Columns.Item].tolist()
        ranks = res['rank'].tolist()
        json_dict = {
            "item_ids":item_ids,
            "ranks":ranks
        }
        return json.dumps(json_dict)
@app.get("/titles-features")
async def titles_features():
    tf_pd = await DataPrepareService.get_titles_features()
    return tf_pd.to_html()

if __name__ == '__main__':
    grpc_thread = threading.Thread(target=serve)
    print(f"Serving grpcs thread 1")
    grpc_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
