import { Card, CardHeader, CardBody, H3, P, Button, Code } from "jadis-ui";
import { useState } from "react";
import { safeExecute, globalPerformanceMonitor } from "~/utils/safeExecution";
import { ErrorBoundary } from "./ErrorBoundary";

interface ExampleCardProps {
  title: string;
  description: string;
  code: string;
  result?: string;
  onRun?: () => Promise<string>;
  category?: string;
  timeout?: number;
  retries?: number;
}

export function ExampleCard({
  title,
  description,
  code,
  result: initialResult,
  onRun,
  category,
  timeout = 10000,
  retries = 1
}: ExampleCardProps) {
  const [result, setResult] = useState(initialResult);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [performanceInfo, setPerformanceInfo] = useState<string | null>(null);

  const handleRun = async () => {
    if (!onRun) return;

    setIsRunning(true);
    setError(null);
    setPerformanceInfo(null);

    const executionResult = await safeExecute(
      async () => {
        const { result: output, duration } = await globalPerformanceMonitor.measure(
          `example-${title}`,
          onRun
        );
        return { output, duration };
      },
      {
        timeout,
        retries,
        fallback: () => ({ output: "Execution timed out or failed, but this is expected behavior for demonstration.", duration: 0 })
      }
    );

    setIsRunning(false);

    if (executionResult.success && executionResult.data) {
      setResult(executionResult.data.output);
      if (executionResult.data.duration > 0) {
        setPerformanceInfo(`Execution time: ${executionResult.data.duration.toFixed(2)}ms`);
      }
      if (executionResult.fallbackUsed) {
        setPerformanceInfo(prev => `${prev || ''} (fallback used)`.trim());
      }
    } else {
      setError(executionResult.error || 'Unknown error occurred');
    }
  };

  return (
    <ErrorBoundary>
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <H3>{title}</H3>
              <div className="flex items-center gap-2 mt-1">
                {category && (
                  <span className="inline-block bg-primary/10 text-primary text-xs px-2 py-1 rounded-full">
                    {category}
                  </span>
                )}
                {performanceInfo && (
                  <span className="inline-block bg-blue-50 text-blue-700 text-xs px-2 py-1 rounded-full">
                    {performanceInfo}
                  </span>
                )}
              </div>
            </div>
            {onRun && (
              <Button
                onClick={handleRun}
                disabled={isRunning}
                size="sm"
              >
                {isRunning ? 'Running...' : 'Run'}
              </Button>
            )}
          </div>
          <P className="text-sm text-muted-foreground mt-2">{description}</P>
        </CardHeader>
        <CardBody>
          <div className="space-y-4">
            {/* Code Section */}
            <div>
              <h4 className="text-sm font-semibold mb-2">Code:</h4>
              <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                <Code className="text-sm whitespace-pre-wrap">{code}</Code>
              </div>
            </div>

            {/* Result Section */}
            {(result || error) && (
              <div>
                <h4 className="text-sm font-semibold mb-2">
                  {error ? 'Error:' : 'Result:'}
                </h4>
                <div className={`p-4 rounded-lg overflow-x-auto ${
                  error ? 'bg-destructive/10 border border-destructive/20' : 'bg-green-50 border border-green-200'
                }`}>
                  <Code className={`text-sm whitespace-pre-wrap ${
                    error ? 'text-destructive' : 'text-green-700'
                  }`}>
                    {error || result}
                  </Code>
                </div>
              </div>
            )}
          </div>
        </CardBody>
      </Card>
    </ErrorBoundary>
  );
}