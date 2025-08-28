import { Divider, Space, Tag } from "antd";

export const TranslationResult = ({ result }) => {
    if (!result) return null;

    return (
        <>
            <Divider>Результаты перевода</Divider>
            <Space direction="vertical">
                {Object.entries(result).map(([lang, translation]) => (
                    <Tag key={lang} color="blue">
                        <b>{lang}</b>: {translation}
                    </Tag>
                ))}
            </Space>
        </>
    );
};
