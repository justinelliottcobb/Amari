import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { MantineProvider, createTheme } from '@mantine/core';
import { MemoryRouter } from 'react-router-dom';

const theme = createTheme({
  primaryColor: 'cyan',
});

interface AllProvidersProps {
  children: React.ReactNode;
  initialEntries?: string[];
}

function AllProviders({ children, initialEntries = ['/'] }: AllProvidersProps) {
  return (
    <MemoryRouter initialEntries={initialEntries}>
      <MantineProvider theme={theme} defaultColorScheme="dark">
        {children}
      </MantineProvider>
    </MemoryRouter>
  );
}

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  initialEntries?: string[];
}

function customRender(
  ui: ReactElement,
  options?: CustomRenderOptions
) {
  const { initialEntries, ...renderOptions } = options || {};

  return render(ui, {
    wrapper: ({ children }) => (
      <AllProviders initialEntries={initialEntries}>{children}</AllProviders>
    ),
    ...renderOptions,
  });
}

// Re-export everything
export * from '@testing-library/react';
export { customRender as render };
