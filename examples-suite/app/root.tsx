import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  isRouteErrorResponse,
  useRouteError,
} from "@remix-run/react";
import type { LinksFunction } from "@remix-run/node";
import { ThemeProvider, Card, CardHeader, CardBody, Button } from "jadis-ui";
import { ErrorBoundary as ComponentErrorBoundary } from "./components/ErrorBoundary";
import "jadis-ui/styles";

export const links: LinksFunction = () => [
  { rel: "preconnect", href: "https://fonts.googleapis.com" },
  {
    rel: "preconnect",
    href: "https://fonts.gstatic.com",
    crossOrigin: "anonymous",
  },
  {
    rel: "stylesheet",
    href: "https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap",
  },
];

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        <ThemeProvider defaultTheme="terminal">
          {children}
        </ThemeProvider>
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

export default function App() {
  return (
    <ComponentErrorBoundary>
      <Outlet />
    </ComponentErrorBoundary>
  );
}

export function ErrorBoundary() {
  const error = useRouteError();

  if (isRouteErrorResponse(error)) {
    return (
      <html lang="en">
        <head>
          <meta charSet="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Error {error.status}</title>
          <Meta />
          <Links />
        </head>
        <body>
          <ThemeProvider defaultTheme="terminal">
            <div className="min-h-screen flex items-center justify-center p-8">
              <Card className="max-w-md w-full">
                <CardHeader>
                  <h1 className="text-2xl font-bold text-destructive">
                    {error.status} Error
                  </h1>
                </CardHeader>
                <CardBody>
                  <div className="space-y-4">
                    <p className="text-muted-foreground">
                      {error.status === 404
                        ? "The page you're looking for doesn't exist."
                        : error.statusText || "An error occurred"}
                    </p>
                    <Button
                      onClick={() => window.location.href = '/'}
                      className="w-full"
                    >
                      Go Home
                    </Button>
                  </div>
                </CardBody>
              </Card>
            </div>
          </ThemeProvider>
          <Scripts />
        </body>
      </html>
    );
  }

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Application Error</title>
        <Meta />
        <Links />
      </head>
      <body>
        <ThemeProvider defaultTheme="terminal">
          <div className="min-h-screen flex items-center justify-center p-8">
            <Card className="max-w-md w-full border-destructive bg-destructive/5">
              <CardHeader>
                <h1 className="text-2xl font-bold text-destructive">
                  Application Error
                </h1>
              </CardHeader>
              <CardBody>
                <div className="space-y-4">
                  <p className="text-muted-foreground">
                    An unexpected error occurred. Please try refreshing the page.
                  </p>
                  <details className="text-xs">
                    <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                      Error details
                    </summary>
                    <pre className="mt-2 p-2 bg-muted rounded text-xs overflow-auto">
                      {error instanceof Error ? error.message : String(error)}
                    </pre>
                  </details>
                  <div className="flex gap-2">
                    <Button
                      onClick={() => window.location.reload()}
                      variant="outline"
                      className="flex-1"
                    >
                      Refresh
                    </Button>
                    <Button
                      onClick={() => window.location.href = '/'}
                      className="flex-1"
                    >
                      Go Home
                    </Button>
                  </div>
                </div>
              </CardBody>
            </Card>
          </div>
        </ThemeProvider>
        <Scripts />
      </body>
    </html>
  );
}