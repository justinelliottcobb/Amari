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
    <Card style={className ? {} : undefined}>
      <CardBody>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '1.5rem',
              height: '1.5rem',
              border: '2px solid transparent',
              borderBottom: '2px solid var(--primary)',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }}></div>
            <span style={{ fontSize: '0.875rem', color: 'var(--muted-foreground)' }}>{message}</span>
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
    <Card style={className ? {} : undefined}>
      <CardBody>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '3rem 0', textAlign: 'center' }}>
          <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '0.5rem' }}>{title}</h3>
          {description && (
            <p style={{ fontSize: '0.875rem', color: 'var(--muted-foreground)', marginBottom: '1rem', maxWidth: '28rem' }}>
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
    <Card style={{ borderColor: 'var(--destructive)', backgroundColor: 'rgba(239, 68, 68, 0.05)' }}>
      <CardHeader>
        <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: 'var(--destructive)' }}>
          Network Error
        </h3>
      </CardHeader>
      <CardBody>
        <p style={{ fontSize: '0.875rem', color: 'var(--muted-foreground)', marginBottom: '1rem' }}>{message}</p>
        {onRetry && (
          <button
            onClick={onRetry}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'var(--primary)',
              color: 'var(--primary-foreground)',
              borderRadius: '0.375rem',
              fontSize: '0.875rem',
              border: 'none',
              cursor: 'pointer',
              transition: 'background-color 0.2s'
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--primary)/90'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--primary)'}
          >
            Retry
          </button>
        )}
      </CardBody>
    </Card>
  );
}