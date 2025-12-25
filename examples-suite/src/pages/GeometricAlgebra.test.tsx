import { describe, it, expect } from 'vitest';
import { render, screen, waitFor } from '../test/test-utils';
import { GeometricAlgebra } from './GeometricAlgebra';

describe('GeometricAlgebra Page', () => {
  it('renders loading state or page title', async () => {
    render(<GeometricAlgebra />);

    // The page may show loading state initially due to WASM loading
    const hasLoading = screen.queryByText(/Loading WASM/i);
    const hasTitle = screen.queryByRole('heading', { name: /Geometric Algebra/i });

    // Either loading or the title should be present
    expect(hasLoading || hasTitle).toBeTruthy();
  });

  it('eventually shows content or loading state', async () => {
    render(<GeometricAlgebra />);

    // Wait a bit for state updates
    await waitFor(() => {
      // Either we see loading state or actual content
      const hasLoading = screen.queryByText(/Loading/i);
      const hasContent = screen.queryByText(/Geometric/i);
      expect(hasLoading || hasContent).toBeTruthy();
    }, { timeout: 1000 });
  });
});
