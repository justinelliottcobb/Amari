import { Card, Loader, Text, Stack, Box, Button, Title } from "@mantine/core";

interface LoadingStateProps {
  message?: string;
}

export function LoadingState({
  message = "Loading..."
}: LoadingStateProps) {
  return (
    <Card withBorder>
      <Card.Section inheritPadding py="xl">
        <Stack align="center" gap="md">
          <Loader color="cyan" />
          <Text size="sm" c="dimmed">{message}</Text>
        </Stack>
      </Card.Section>
    </Card>
  );
}

interface EmptyStateProps {
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export function EmptyState({
  title,
  description,
  action
}: EmptyStateProps) {
  return (
    <Card withBorder>
      <Card.Section inheritPadding py="xl">
        <Stack align="center" gap="md" style={{ textAlign: 'center' }}>
          <Title order={4}>{title}</Title>
          {description && (
            <Text size="sm" c="dimmed" maw={400}>
              {description}
            </Text>
          )}
          {action}
        </Stack>
      </Card.Section>
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
    <Card
      withBorder
      style={{
        borderColor: 'var(--mantine-color-red-6)',
        backgroundColor: 'rgba(239, 68, 68, 0.05)'
      }}
    >
      <Card.Section withBorder inheritPadding py="sm">
        <Title order={4} c="red">Network Error</Title>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <Stack gap="md">
          <Text size="sm" c="dimmed">{message}</Text>
          {onRetry && (
            <Button onClick={onRetry} size="sm">
              Retry
            </Button>
          )}
        </Stack>
      </Card.Section>
    </Card>
  );
}
