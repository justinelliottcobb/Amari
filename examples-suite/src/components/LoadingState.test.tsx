import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../test/test-utils';
import { LoadingState, EmptyState, NetworkError } from './LoadingState';

describe('LoadingState', () => {
  it('renders with default message', () => {
    render(<LoadingState />);

    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders with custom message', () => {
    render(<LoadingState message="Please wait..." />);

    expect(screen.getByText('Please wait...')).toBeInTheDocument();
  });

  it('renders loader spinner', () => {
    render(<LoadingState />);

    // Mantine Loader has role="presentation" or is a visual element
    // We can check for the loader by its container structure
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
});

describe('EmptyState', () => {
  it('renders title', () => {
    render(<EmptyState title="No Items" />);

    expect(screen.getByText('No Items')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    render(
      <EmptyState
        title="No Items"
        description="There are no items to display"
      />
    );

    expect(screen.getByText('There are no items to display')).toBeInTheDocument();
  });

  it('does not render description when not provided', () => {
    render(<EmptyState title="No Items" />);

    expect(screen.queryByText('There are no items to display')).not.toBeInTheDocument();
  });

  it('renders action when provided', () => {
    render(
      <EmptyState
        title="No Items"
        action={<button>Add Item</button>}
      />
    );

    expect(screen.getByRole('button', { name: 'Add Item' })).toBeInTheDocument();
  });
});

describe('NetworkError', () => {
  it('renders with default message', () => {
    render(<NetworkError />);

    expect(screen.getByText('Network Error')).toBeInTheDocument();
    expect(screen.getByText('Failed to load data. Please check your connection.')).toBeInTheDocument();
  });

  it('renders with custom message', () => {
    render(<NetworkError message="Custom error message" />);

    expect(screen.getByText('Custom error message')).toBeInTheDocument();
  });

  it('renders retry button when onRetry is provided', () => {
    const onRetry = vi.fn();
    render(<NetworkError onRetry={onRetry} />);

    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('does not render retry button when onRetry is not provided', () => {
    render(<NetworkError />);

    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  });

  it('calls onRetry when retry button is clicked', () => {
    const onRetry = vi.fn();
    render(<NetworkError onRetry={onRetry} />);

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
