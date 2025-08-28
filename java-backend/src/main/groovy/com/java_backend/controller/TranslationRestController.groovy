package com.java_backend.controller


import com.java_backend.dto.TranslationRequestDto;
import com.java_backend.service.TranslationService;
import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping("/api/translate")
class TranslationRestController {

    private final TranslationService translationService;

    TranslationRestController(TranslationService translationService) {
        this.translationService = translationService;
    }

    @PostMapping
    Map<String, String> translate(@RequestBody TranslationRequestDto dto) {
        String fromLang = dto.getFromLang() != null ? dto.getFromLang() : "en"; // дефолт если не пришёл
        return translationService.translate(dto.getText(), fromLang, dto.getLangs());
    }
}


