import { useState } from "react";
import {Input, Button, Space, message, Select} from "antd";
import {LANGUAGE_OPTIONS} from "../data/languageOptions";

const { TextArea } = Input;

export const TranslationForm = ({ onTranslate, loading }) => {
    const [text, setText] = useState("");
    const [langs, setLangs] = useState([]);
    const [fromLang, setFromLang] = useState("en");

    const handleClick = () => {
        if (!text.trim()) {
            message.warning("Введите текст");
            return;
        }
        if (!fromLang) {
            message.warning("Выберите язык исходного текста");
            return;
        }
        if (langs.length === 0) {
            message.warning("Выберите хотя бы один язык для перевода");
            return;
        }
        onTranslate(text, fromLang, langs);
    };
    return (
        <Space direction="vertical" style={{ width: "100%" }}>
            <Space style={{ width: "100%" }}>
                <TextArea
                    rows={4}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter text for translation"
                    style={{ flex: 1 }}
                />
                <Select
                    style={{ width: 150 }}
                    value={fromLang}
                    onChange={(value) => {
                        setFromLang(value);
                        setLangs([]);
                    }}
                    options={LANGUAGE_OPTIONS}
                    placeholder="Исходный язык"
                />
            </Space>

            <Select
                mode="multiple"
                allowClear
                style={{ width: "100%" }}
                placeholder="Choose languages for translation"
                value={langs}
                onChange={setLangs}
                options={LANGUAGE_OPTIONS.filter(el=>el.value !==fromLang)}
            />

            <Button type="primary" onClick={handleClick} loading={loading}>
                Перевести
            </Button>
        </Space>
    );
};
