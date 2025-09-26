import { Component, ReactNode } from "react";
import { Card, CardHeader, CardBody, Button } from "jadis-ui";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: undefined });
    this.props.onReset?.();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <Card style={{ borderColor: 'var(--destructive)', backgroundColor: 'rgba(239, 68, 68, 0.05)' }}>
          <CardHeader>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: 'var(--destructive)' }}>
              Something went wrong
            </h3>
          </CardHeader>
          <CardBody>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <p style={{ fontSize: '0.875rem', color: 'var(--muted-foreground)' }}>
                An unexpected error occurred while rendering this component.
              </p>

              {this.state.error && (
                <details style={{ fontSize: '0.75rem' }}>
                  <summary style={{
                    cursor: 'pointer',
                    color: 'var(--muted-foreground)',
                    transition: 'color 0.2s'
                  }}>
                    Error details
                  </summary>
                  <pre style={{
                    marginTop: '0.5rem',
                    padding: '0.5rem',
                    backgroundColor: 'var(--muted)',
                    borderRadius: '0.25rem',
                    fontSize: '0.75rem',
                    overflow: 'auto',
                    whiteSpace: 'pre-wrap'
                  }}>
                    {this.state.error.message}
                    {this.state.error.stack && `\n\n${this.state.error.stack}`}
                  </pre>
                </details>
              )}

              <Button
                onClick={this.handleReset}
                size="sm"
                variant="outline"
              >
                Try again
              </Button>
            </div>
          </CardBody>
        </Card>
      );
    }

    return this.props.children;
  }
}

export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  fallback?: ReactNode,
  onReset?: () => void
) {
  return function WrappedComponent(props: P) {
    return (
      <ErrorBoundary fallback={fallback} onReset={onReset}>
        <Component {...props} />
      </ErrorBoundary>
    );
  };
}