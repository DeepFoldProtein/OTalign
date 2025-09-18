'use client';

import { useState } from 'react';
import { SubmissionData } from '@/lib/types';

export default function SubmissionForm() {
  const [formData, setFormData] = useState<SubmissionData>({
    model: '',
    type: '',
    description: '',
    paper_url: '',
    code_url: '',
    malidup_f1: undefined,
    malisam_f1: undefined,
    sabmark_sup_recall: undefined,
    sabmark_twi_recall: undefined,
    organization: '',
    date_submitted: new Date().toISOString().split('T')[0],
  });

  const [generatedJson, setGeneratedJson] = useState<string>('');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value === '' ? (name.includes('_f1') || name.includes('_recall') ? undefined : '') : 
               (name.includes('_f1') || name.includes('_recall') ? parseFloat(value) : value)
    }));
  };

  const generateSubmission = () => {
    const submissionData = {
      ...formData,
      date_submitted: new Date().toISOString().split('T')[0],
    };
    setGeneratedJson(JSON.stringify(submissionData, null, 2));
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(generatedJson);
      alert('JSON copied to clipboard!');
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Instructions */}
      <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-lg border border-blue-200 dark:border-blue-800">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-3">
          How to Submit Your Method
        </h3>
        <ol className="list-decimal list-inside space-y-2 text-blue-800 dark:text-blue-200">
          <li><strong>Run Evaluation:</strong> Execute your alignment method on our benchmark datasets</li>
          <li><strong>Fill Form:</strong> Complete the form below with your method details and performance metrics</li>
          <li><strong>Generate JSON:</strong> Click "Generate Submission JSON" to create the submission file</li>
          <li><strong>Submit:</strong> Create a Pull Request to our GitHub repository with the JSON file</li>
        </ol>
      </div>

      {/* Form */}
      <div className="bg-white dark:bg-gray-800 p-8 rounded-lg border border-gray-200 dark:border-gray-700">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          Submit Your Method
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Basic Information */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Basic Information</h3>
            
            <div>
              <label htmlFor="model" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Model Name *
              </label>
              <input
                type="text"
                id="model"
                name="model"
                value={formData.model}
                onChange={handleInputChange}
                placeholder="YourMethod-v1.0"
                required
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
            </div>

            <div>
              <label htmlFor="type" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Method Type *
              </label>
              <select
                id="type"
                name="type"
                value={formData.type}
                onChange={handleInputChange}
                required
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              >
                <option value="">Select type...</option>
                <option value="Traditional">Traditional</option>
                <option value="PLM-based">PLM-based</option>
                <option value="OTalign">OTalign</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div>
              <label htmlFor="organization" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Organization *
              </label>
              <input
                type="text"
                id="organization"
                name="organization"
                value={formData.organization}
                onChange={handleInputChange}
                placeholder="Your University/Company"
                required
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
            </div>

            <div>
              <label htmlFor="description" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Description *
              </label>
              <textarea
                id="description"
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Brief description of your method..."
                rows={3}
                required
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
            </div>
          </div>

          {/* URLs and Metrics */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Links & Performance</h3>
            
            <div>
              <label htmlFor="code_url" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Code URL *
              </label>
              <input
                type="url"
                id="code_url"
                name="code_url"
                value={formData.code_url}
                onChange={handleInputChange}
                placeholder="https://github.com/..."
                required
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
            </div>

            <div>
              <label htmlFor="paper_url" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Paper URL (optional)
              </label>
              <input
                type="url"
                id="paper_url"
                name="paper_url"
                value={formData.paper_url}
                onChange={handleInputChange}
                placeholder="https://arxiv.org/abs/..."
                className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
            </div>

            {/* Performance Metrics */}
            <div className="space-y-3">
              <h4 className="text-md font-medium text-gray-900 dark:text-white">Performance Metrics</h4>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label htmlFor="malidup_f1" className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    MALIDUP F1
                  </label>
                  <input
                    type="number"
                    id="malidup_f1"
                    name="malidup_f1"
                    value={formData.malidup_f1 || ''}
                    onChange={handleInputChange}
                    step="0.0001"
                    min="0"
                    max="1"
                    placeholder="0.0000"
                    className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="malisam_f1" className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    MALISAM F1
                  </label>
                  <input
                    type="number"
                    id="malisam_f1"
                    name="malisam_f1"
                    value={formData.malisam_f1 || ''}
                    onChange={handleInputChange}
                    step="0.0001"
                    min="0"
                    max="1"
                    placeholder="0.0000"
                    className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="sabmark_sup_recall" className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    SABmark (sup) Recall
                  </label>
                  <input
                    type="number"
                    id="sabmark_sup_recall"
                    name="sabmark_sup_recall"
                    value={formData.sabmark_sup_recall || ''}
                    onChange={handleInputChange}
                    step="0.0001"
                    min="0"
                    max="1"
                    placeholder="0.0000"
                    className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  />
                </div>

                <div>
                  <label htmlFor="sabmark_twi_recall" className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                    SABmark (twi) Recall
                  </label>
                  <input
                    type="number"
                    id="sabmark_twi_recall"
                    name="sabmark_twi_recall"
                    value={formData.sabmark_twi_recall || ''}
                    onChange={handleInputChange}
                    step="0.0001"
                    min="0"
                    max="1"
                    placeholder="0.0000"
                    className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Generate Button */}
        <div className="mt-8">
          <button
            onClick={generateSubmission}
            disabled={!formData.model || !formData.type || !formData.description || !formData.code_url || !formData.organization}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            Generate Submission JSON
          </button>
        </div>
      </div>

      {/* Generated JSON */}
      {generatedJson && (
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Generated Submission JSON
            </h3>
            <button
              onClick={copyToClipboard}
              className="bg-green-600 text-white py-2 px-4 rounded-md text-sm font-medium hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-colors"
            >
              Copy to Clipboard
            </button>
          </div>
          <pre className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md overflow-x-auto text-sm font-mono border border-gray-200 dark:border-gray-600">
            <code className="text-gray-900 dark:text-gray-100">{generatedJson}</code>
          </pre>
          <p className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            Save this JSON to a file and submit it via a Pull Request to our GitHub repository.
          </p>
        </div>
      )}
    </div>
  );
}
