import { Card, CardHeader, CardBody, H3, P, Button, CodeBlock, StatusBadge } from "jadis-ui";
import { useState } from "react";
import { safeExecute, globalPerformanceMonitor } from "../utils/safeExecution";
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
      <Card style={{ marginBottom: '1.5rem' }}>
        <CardHeader>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <H3>{title}</H3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.25rem' }}>
                {category && (
                  <StatusBadge variant="info">
                    {category}
                  </StatusBadge>
                )}
                {performanceInfo && (
                  <StatusBadge variant="success">
                    {performanceInfo}
                  </StatusBadge>
                )}
              </div>
            </div>
            {onRun && (
              <Button
                onClick={handleRun}
                disabled={isRunning}
              >
                {isRunning ? 'Running...' : 'Run'}
              </Button>
            )}
          </div>
          <P style={{ fontSize: '0.875rem', marginTop: '0.5rem', opacity: 0.7 }}>{description}</P>
        </CardHeader>
        <CardBody>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {/* Code Section */}
            <div>
              <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>Code:</h4>
              <CodeBlock
                language="javascript"
                showLineNumbers={true}
                showCopyButton={true}
                style={{ width: '100%' }}
              >
                {code}
              </CodeBlock>
            </div>

            {/* Result Section */}
            {(result || error) && (
              <div>
                <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                  {error ? 'Error:' : 'Result:'}
                </h4>
                <CodeBlock
                  language="text"
                  variant={error ? 'error' : 'success'}
                  style={{ width: '100%' }}
                >
                  {error || result}
                </CodeBlock>
              </div>
            )}
          </div>
        </CardBody>
      </Card>
    </ErrorBoundary>
  );
}