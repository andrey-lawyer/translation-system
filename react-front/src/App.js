import './App.css';

import { Typography } from "antd";
import { useTranslation } from "./hooks/useTranslation";
import { TranslationForm } from "./components/TranslationForm";
import { TranslationResult } from "./components/TranslationResult";


function App() {
  const { result, translate, loading } = useTranslation();

  return (
      <div style={{ maxWidth: 600, margin: "40px auto" }}>
        <Typography.Title level={2}>🈺 Translation</Typography.Title>
        <TranslationForm onTranslate={translate} loading={loading} />
        <TranslationResult result={result} />
      </div>
  );
}

export default App;
