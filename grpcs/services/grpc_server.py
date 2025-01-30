import grpc
from concurrent import futures
import time

from sqlalchemy.orm import Session

from core.database import SessionLocal
from grpcs.mappers.rec_mappers import UserMapper
from grpcs.protos import rec_pb2, rec_pb2_grpc
from  models import RawUsers

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class RecService(rec_pb2_grpc.RecServiceServicer):
    def GetUser(self, request, context):
        db: Session = SessionLocal()
        try:
            user = db.query(RawUsers).filter_by(id=1).first()
            user_response = UserMapper.to_grpc(user)
            return user_response
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return rec_pb2.UserResponse


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rec_pb2_grpc.add_RecServiceServicer_to_server(RecService(), server)
    server.add_insecure_port('[::]:50051')
    print(server)
    server.start()
    print('gRPC server running on port 50051')
    server.wait_for_termination()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
    print('gRPC server running on port 500511')

if __name__ == '__main__':
    serve()
