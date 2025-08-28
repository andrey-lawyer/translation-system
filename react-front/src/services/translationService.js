import axios from "axios";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8080/api/translate";

export const translateText = async (text,fromLang, langs) => {
    const {data} = await axios.post(API_URL, {
        text,
        fromLang,
        langs,
    });
    return data;
};
