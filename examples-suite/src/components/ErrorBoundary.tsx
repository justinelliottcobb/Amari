import { Component, ReactNode } from "react";
import { Card, Text, Button, Box, Stack } from "@mantine/core";

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
        <Card
          withBorder
          style={{
            borderColor: 'var(--mantine-color-red-6)',
            backgroundColor: 'rgba(239, 68, 68, 0.05)'
          }}
        >
          <Card.Section withBorder inheritPadding py="sm">
            <Text fw={600} size="lg" c="red">
              Something went wrong
            </Text>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Stack gap="md">
              <Text size="sm" c="dimmed">
                An unexpected error occurred while rendering this component.
              </Text>

              {this.state.error && (
                <details style={{ fontSize: '0.75rem' }}>
                  <summary style={{
                    cursor: 'pointer',
                    color: 'var(--mantine-color-dimmed)'
                  }}>
                    Error details
                  </summary>
                  <Box
                    mt="xs"
                    p="xs"
                    bg="dark.7"
                    style={{ borderRadius: 'var(--mantine-radius-sm)' }}
                  >
                    <Text
                      size="xs"
                      ff="monospace"
                      style={{ whiteSpace: 'pre-wrap', overflow: 'auto' }}
                    >
                      {this.state.error.message}
                      {this.state.error.stack && `\n\n${this.state.error.stack}`}
                    </Text>
                  </Box>
                </details>
              )}

              <Button
                onClick={this.handleReset}
                size="sm"
                variant="outline"
              >
                Try again
              </Button>
            </Stack>
          </Card.Section>
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
