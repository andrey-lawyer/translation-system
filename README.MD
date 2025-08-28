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

### Generate protobuf code (from the repository root)

- For Go (go-translator):

# Install plugins (once)
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
# Ensure that %GOPATH%\bin is in PATH (Windows)

# Generate code

```bash
protoc --go_out=go-translator/translatorpb --go-grpc_out=go-translator/translatorpb proto/translation.proto
```
# Alternative with relative paths:

```bash
protoc --go_out=paths=source_relative:go-translator/translatorpb --go-grpc_out=paths=source_relative:go-translator/translatorpb proto/translation.proto
```

- For Java (java-backend):

# Generate code

```bash
./java-backend/gradlew -p java-backend generateProto
```

or

```bash
cd java-backend
./gradlew generateProto
```


# Run Go Translator
cd go-translator
go run main.go

# In a new terminal window/tab, run Java Backend
cd java-backend
./gradlew bootRun

---

## Docker: run the full project

Requirements: Docker Desktop (or compatible) with docker compose.

1) From repository root:
```bash
docker compose up --build
```

2) After start, check:
- React (UI): http://localhost:3000
- Java REST API: http://localhost:8080/api/translate
- Go gRPC (inside docker network): go-translator:6565 (Java uses this name)

3) Stop:
```bash
docker compose down
```

4) Useful commands:
```bash
docker compose build java-backend
docker compose logs -f java-backend
```

What compose does:
- Runs 3 containers: react-front (Nginx, port 3000), java-backend (Spring Boot, port 8080), go-translator (gRPC, port 6565).
- react-front proxies /api/ to java-backend:8080/api/.
- java-backend calls Go using GO_TRANSLATOR_HOST=go-translator and GO_TRANSLATOR_PORT=6565.

Environment variables (see docker-compose.yml):
- GO_TRANSLATOR_HOST (default go-translator)
- GO_TRANSLATOR_PORT (default 6565)
- FRONTEND_ORIGIN (default http://localhost:3000, for CORS)
- REACT_APP_API_URL (frontend build arg, default /api/translate)

Notes:

- Java protobuf generation happens inside the java-backend image during build. Go protobuf code already exists in go-translator/translatorpb.


## How to Test

### gRPC (Go Translator)

Use `grpcurl` to call the Go service directly:

```bash
grpcurl -plaintext \
  -d '{"text":"hello world","fromLang":"en","langs":["ru","uk"]}' \
  localhost:6565 translator.TranslatorService/Translate
```

Expected response:

```json
{
  "translations": {
    "en": "hello world",
    "ru": "привет мир",
    "uk": "привіт світ"
  }
}
```

### REST (Java Backend)

Use `curl` to hit the REST endpoint:

```bash
curl -X POST http://localhost:8080/api/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world",
    "fromLang": "en",
    "langs": ["ru", "uk"]
  }'
```
