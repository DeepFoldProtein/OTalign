interface LoadingSpinnerProps {
  message?: string;
}

export default function LoadingSpinner({
  message = "Loading...",
}: LoadingSpinnerProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
      <div className="text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-[var(--toss-blue)] border-t-transparent mx-auto mb-4"></div>
        <p className="text-[var(--toss-light-gray)] text-sm">{message}</p>
      </div>
    </div>
  );
}

