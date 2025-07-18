package main

import (
	"context"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	pb "go-translator/translatorpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

// server реализует интерфейс TranslatorServiceServer
type server struct {
	pb.UnimplementedTranslatorServiceServer
}

func (s *server) Translate(ctx context.Context, req *pb.TranslateRequest) (*pb.TranslateResponse, error) {
	englishText := req.GetText()
	langs := req.GetLangs()

	log.Printf("Received translation request: text='%s', langs=%v", englishText, langs)

	result := map[string]string{
		"en": englishText,
	}

	for _, lang := range langs {
		switch lang {
		case "ru":
			result[lang] = "привет мир"
		case "uk":
			result[lang] = "привіт світ"
		default:
			result[lang] = "[unknown language: " + lang + "]"
		}
	}

	log.Printf("Sending translation response: %v", result)

	return &pb.TranslateResponse{Translations: result}, nil
}

func main() {
	lis, err := net.Listen("tcp", ":6565")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterTranslatorServiceServer(grpcServer, &server{})
	reflection.Register(grpcServer)

	go func() {
		log.Println("gRPC server listening on :6565")
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("failed to serve: %v", err)
		}
	}()

	// graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop

	log.Println("Stopping gRPC server...")
	grpcServer.GracefulStop()
	log.Println("Server stopped")
}
