interface ErrorDisplayProps {
  message: string;
}

export default function ErrorDisplay({ message }: ErrorDisplayProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--page)] px-6">
      <div className="text-center max-w-sm">
        <div className="text-[15px] font-semibold text-[var(--ink)] mb-1.5">
          Something went wrong
        </div>
        <p className="text-[14px] text-[var(--ink-2)]">{message}</p>
      </div>
    </div>
  );
}
