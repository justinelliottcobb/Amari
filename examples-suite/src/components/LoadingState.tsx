import { Card, CardHeader, CardBody } from "jadis-ui";

interface LoadingStateProps {
  message?: string;
  className?: string;
}

export function LoadingState({
  message = "Loading...",
  className = ""
}: LoadingStateProps) {
  return (
    <Card className={className}>
      <CardBody>
        <div className="flex items-center justify-center py-8">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
            <span className="text-sm text-muted-foreground">{message}</span>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

interface EmptyStateProps {
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export function EmptyState({
  title,
  description,
  action,
  className = ""
}: EmptyStateProps) {
  return (
    <Card className={className}>
      <CardBody>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <h3 className="text-lg font-semibold mb-2">{title}</h3>
          {description && (
            <p className="text-sm text-muted-foreground mb-4 max-w-md">
              {description}
            </p>
          )}
          {action}
        </div>
      </CardBody>
    </Card>
  );
}

interface NetworkErrorProps {
  onRetry?: () => void;
  message?: string;
}

export function NetworkError({
  onRetry,
  message = "Failed to load data. Please check your connection."
}: NetworkErrorProps) {
  return (
    <Card className="border-destructive bg-destructive/5">
      <CardHeader>
        <h3 className="text-lg font-semibold text-destructive">
          Network Error
        </h3>
      </CardHeader>
      <CardBody>
        <p className="text-sm text-muted-foreground mb-4">{message}</p>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm hover:bg-primary/90"
          >
            Retry
          </button>
        )}
      </CardBody>
    </Card>
  );
}