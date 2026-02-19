# RobotEnv Protocol Buffers

This directory contains the Protocol Buffer definitions for the RobotEnv gRPC service.

## Files

- `robotenv.proto` - Protocol buffer definition (source of truth)
- `robotenv_pb2.py` - Generated message classes (auto-generated)
- `robotenv_pb2_grpc.py` - Generated gRPC service stubs (auto-generated)

## Compiling

Run the compilation script:

```bash
bash proto/compile_proto.sh
```

This generates:
- `robotenv_pb2.py` - Message classes (ObservationSpec, ResetRequest, StepResponse, etc.)
- `robotenv_pb2_grpc.py` - gRPC service stubs (RobotEnvServicer, RobotEnvStub)

## Regenerating

Re-run compilation after any changes to `robotenv.proto`:

```bash
bash proto/compile_proto.sh
```

## Requirements

Make sure `grpcio-tools` is installed:

```bash
pip install grpcio-tools
```

## Usage

### Server Implementation

```python
from proto import robotenv_pb2_grpc
from proto import robotenv_pb2

class MyRobotEnvService(robotenv_pb2_grpc.RobotEnvServicer):
    def Step(self, request, context):
        # Implement step logic
        return robotenv_pb2.StepResponse(...)
```

### Client Usage

```python
import grpc
from proto import robotenv_pb2_grpc
from proto import robotenv_pb2

channel = grpc.insecure_channel('localhost:50051')
stub = robotenv_pb2_grpc.RobotEnvStub(channel)

response = stub.HealthCheck(robotenv_pb2.HealthCheckRequest())
print(response.status)  # "HEALTHY"
```
