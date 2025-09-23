import { Card, CardHeader, CardBody, H3, P, Button, Code } from "jadis";
import { useState } from "react";

interface ExampleCardProps {
  title: string;
  description: string;
  code: string;
  result?: string;
  onRun?: () => Promise<string>;
  category?: string;
}

export function ExampleCard({
  title,
  description,
  code,
  result: initialResult,
  onRun,
  category
}: ExampleCardProps) {
  const [result, setResult] = useState(initialResult);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    if (!onRun) return;

    setIsRunning(true);
    setError(null);
    try {
      const output = await onRun();
      setResult(output);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <Card className="mb-6">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <H3>{title}</H3>
            {category && (
              <span className="inline-block bg-primary/10 text-primary text-xs px-2 py-1 rounded-full mt-1">
                {category}
              </span>
            )}
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
  );
}