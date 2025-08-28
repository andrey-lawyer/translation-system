package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"go-translator/internal/server"
)

func main() {
	grpcServer, err := server.StartGRPCServer(":6565")
	if err != nil {
		log.Fatalf("failed to start server: %v", err)
	}

	// graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	log.Println("Stopping gRPC server...")
	grpcServer.GracefulStop()
	log.Println("Server stopped")
}
