import { describe, it, expect } from 'vitest';
import { render, screen } from '../test/test-utils';
import { Calculus } from './Calculus';

describe('Calculus Page', () => {
  it('renders the page title', () => {
    render(<Calculus />);

    expect(screen.getByRole('heading', { name: 'Differential Calculus' })).toBeInTheDocument();
  });

  it('renders the page description', () => {
    render(<Calculus />);

    const descElements = screen.getAllByText(/differential/i);
    expect(descElements.length).toBeGreaterThan(0);
  });

  it('renders gradient section', () => {
    render(<Calculus />);

    const gradientElements = screen.getAllByText(/Gradient/i);
    expect(gradientElements.length).toBeGreaterThan(0);
  });

  it('renders divergence section', () => {
    render(<Calculus />);

    const divergenceElements = screen.getAllByText(/Divergence/i);
    expect(divergenceElements.length).toBeGreaterThan(0);
  });

  it('renders curl section', () => {
    render(<Calculus />);

    const curlElements = screen.getAllByText(/Curl/i);
    expect(curlElements.length).toBeGreaterThan(0);
  });

  it('renders integration section', () => {
    render(<Calculus />);

    const integrationElements = screen.getAllByText(/Integration/i);
    expect(integrationElements.length).toBeGreaterThan(0);
  });

  it('renders code examples', () => {
    render(<Calculus />);

    expect(screen.getAllByText(/Code:/i).length).toBeGreaterThan(0);
  });
});
