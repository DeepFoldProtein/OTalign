import { useState } from "react";
import { SubmissionData } from "@/lib/types";

interface UseSubmissionFormReturn {
  formData: SubmissionData;
  generatedJson: string;
  handleInputChange: (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
    >
  ) => void;
  generateSubmission: () => void;
  copyToClipboard: () => Promise<void>;
  isFormValid: boolean;
}

export function useSubmissionForm(): UseSubmissionFormReturn {
  const [formData, setFormData] = useState<SubmissionData>({
    model: "",
    type: "",
    description: "",
    paper_url: "",
    code_url: "",
    malidup_f1: undefined,
    malisam_f1: undefined,
    sabmark_sup_recall: undefined,
    sabmark_twi_recall: undefined,
    organization: "",
    date_submitted: new Date().toISOString().split("T")[0],
  });

  const [generatedJson, setGeneratedJson] = useState<string>("");

  const handleInputChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement
    >
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]:
        value === ""
          ? name.includes("_f1") || name.includes("_recall")
            ? undefined
            : ""
          : name.includes("_f1") || name.includes("_recall")
          ? parseFloat(value)
          : value,
    }));
  };

  const generateSubmission = () => {
    const submissionData = {
      ...formData,
      date_submitted: new Date().toISOString().split("T")[0],
    };
    setGeneratedJson(JSON.stringify(submissionData, null, 2));
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(generatedJson);
      alert("JSON copied to clipboard!");
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const isFormValid = Boolean(
    formData.model &&
      formData.type &&
      formData.description &&
      formData.code_url &&
      formData.organization
  );

  return {
    formData,
    generatedJson,
    handleInputChange,
    generateSubmission,
    copyToClipboard,
    isFormValid,
  };
}
