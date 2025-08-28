import { useState } from "react";
import { translateText } from "../services/translationService";
import { message } from "antd";

export const useTranslation = () => {
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const translate = async (text, fromLang, langs) => {
        setLoading(true);
        try {
            // передаем все параметры в API / функцию перевода
            const data = await translateText(text, fromLang, langs);
            setResult(data);
        } catch (err) {
            message.error("Error translations");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return {
        result,
        translate,
        loading,
    };
};
