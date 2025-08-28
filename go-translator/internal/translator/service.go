package translator

import (
	"context"
	pb "go-translator/translatorpb"
)

type TranslatorService struct {
	pb.UnimplementedTranslatorServiceServer
}

func NewTranslatorService() *TranslatorService {
	return &TranslatorService{}
}

func (s *TranslatorService) Translate(ctx context.Context, req *pb.TranslateRequest) (*pb.TranslateResponse, error) {
	fromLang := req.GetFromLang()
	if fromLang == "" {
		fromLang = "en" // дефолт
	}
	englishText := req.GetText()

	result := map[string]string{fromLang: englishText}

	for _, lang := range req.GetLangs() {
		translated, err := TranslateMyMemory(ctx, englishText, fromLang, lang)
		if err != nil {
			translated = "[translation error: " + err.Error() + "]"
		}
		result[lang] = translated
	}

	return &pb.TranslateResponse{Translations: result}, nil
}
