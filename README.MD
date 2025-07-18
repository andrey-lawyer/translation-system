# Translation System (Java + Go)

This project implements a simple translation microservice system using gRPC with two components:

- **Java Backend** — main backend service that handles client requests and delegates translation calls to the Go service.
- **Go Translator** — gRPC service that performs mock translations.

---

## Project Structure

translation-system/
├── java-backend/ # Java Spring Boot backend with gRPC client and server
├── go-translator/ # Go gRPC translation service
└── proto/ # Protobuf definitions shared by both services


---

## How it works

1. Client sends a translation request (English text + target languages) to the Java backend.
2. Java backend forwards the request to the Go translator service via gRPC.
3. Go translator returns mock translated texts.
4. Java backend returns the combined translations to the client.

---

## Prerequisites

- Java 17+ and Gradle for the Java backend
- Go 1.18+ for the Go translator
- `protoc` compiler with Go and Java plugins installed
- `grpcurl` or any gRPC client tool to test the services

---

## Setup & Run

### Generate protobuf code

From the project root, run:

```bash
protoc --go_out=. --go-grpc_out=. proto/translation.proto
./gradlew generateProto
```

# Run Go Translator
cd go-translator
go run main.go

# In a new terminal window/tab, run Java Backend
cd java-backend
./gradlew bootRun
