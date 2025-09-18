interface ErrorDisplayProps {
  message: string;
}

export default function ErrorDisplay({ message }: ErrorDisplayProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
      <div className="text-center">
        <div className="w-12 h-12 rounded-full bg-red-50 dark:bg-red-900/20 flex items-center justify-center mx-auto mb-4">
          <span className="text-red-500 text-xl">⚠️</span>
        </div>
        <p className="text-red-600 dark:text-red-400">{message}</p>
      </div>
    </div>
  );
}

