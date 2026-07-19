interface LoadingSpinnerProps {
  message?: string;
}

export default function LoadingSpinner({
  message = "Loading…",
}: LoadingSpinnerProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--page)]">
      <div className="text-center">
        <div className="animate-spin rounded-full h-7 w-7 border-2 border-[var(--accent)] border-t-transparent mx-auto mb-3" />
        <p className="text-[13px] text-[var(--ink-3)]">{message}</p>
      </div>
    </div>
  );
}
