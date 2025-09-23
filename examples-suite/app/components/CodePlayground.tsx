import { useState, useCallback } from "react";
import { Card, CardHeader, CardBody, Button, Code } from "jadis";

interface CodePlaygroundProps {
  initialCode?: string;
  title?: string;
  description?: string;
  height?: string;
  showLineNumbers?: boolean;
  language?: string;
  onRun?: (code: string) => Promise<{ output: string; error?: string }>;
}

export function CodePlayground({
  initialCode = "",
  title = "Code Editor",
  description,
  height = "h-64",
  showLineNumbers = true,
  language = "javascript",
  onRun
}: CodePlaygroundProps) {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  const runCode = useCallback(async () => {
    if (!onRun) return;

    setIsRunning(true);
    setError(null);
    setOutput("");

    try {
      const result = await onRun(code);
      setOutput(result.output);
      if (result.error) {
        setError(result.error);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsRunning(false);
    }
  }, [code, onRun]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Allow tab insertion in textarea
    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.target as HTMLTextAreaElement;
      const start = target.selectionStart;
      const end = target.selectionEnd;

      const newCode = code.substring(0, start) + '  ' + code.substring(end);
      setCode(newCode);

      // Move cursor after the inserted spaces
      setTimeout(() => {
        target.selectionStart = target.selectionEnd = start + 2;
      }, 0);
    }

    // Run code with Ctrl/Cmd + Enter
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runCode();
    }
  };

  const lineNumbers = showLineNumbers && code.split('\n').map((_, i) => i + 1);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            {description && (
              <p className="text-sm text-muted-foreground mt-1">{description}</p>
            )}
          </div>
          {onRun && (
            <div className="flex gap-2">
              <Button
                onClick={runCode}
                disabled={isRunning}
                size="sm"
                title="Run code (Ctrl/Cmd + Enter)"
              >
                {isRunning ? 'Running...' : 'Run'}
              </Button>
              <Button
                onClick={() => setCode(initialCode)}
                variant="outline"
                size="sm"
                title="Reset to initial code"
              >
                Reset
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardBody>
        <div className="relative">
          <div className="flex">
            {showLineNumbers && (
              <div className="select-none pr-3 text-right text-xs text-muted-foreground">
                {lineNumbers && lineNumbers.map(n => (
                  <div key={n}>{n}</div>
                ))}
              </div>
            )}
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              onKeyDown={handleKeyDown}
              className={`flex-1 ${height} p-3 font-mono text-sm bg-muted rounded-lg border border-border focus:ring-2 focus:ring-primary focus:border-primary resize-none`}
              spellCheck={false}
              placeholder="Enter your code here..."
            />
          </div>
        </div>

        {/* Output Section */}
        {(output || error) && (
          <div className="mt-4">
            <h4 className="text-sm font-semibold mb-2">Output:</h4>
            {error ? (
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-3">
                <Code className="text-sm text-destructive whitespace-pre-wrap">
                  {error}
                </Code>
              </div>
            ) : (
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <Code className="text-sm text-green-700 whitespace-pre-wrap">
                  {output}
                </Code>
              </div>
            )}
          </div>
        )}
      </CardBody>
    </Card>
  );
}

// Mini inline playground for small code snippets
export function InlinePlayground({
  code,
  language = "javascript"
}: {
  code: string;
  language?: string;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [output, setOutput] = useState<string>("");

  const runInlineCode = async () => {
    try {
      // Simple evaluation for demonstration
      const logs: string[] = [];
      const consoleProxy = {
        log: (...args: any[]) => {
          logs.push(args.join(' '));
        }
      };

      const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
      const fn = new AsyncFunction('console', code);
      const result = await fn(consoleProxy);

      setOutput(logs.join('\n') + (result ? `\nReturned: ${JSON.stringify(result)}` : ''));
    } catch (err) {
      setOutput(`Error: ${err}`);
    }
  };

  return (
    <div className="my-3 border border-border rounded-lg overflow-hidden">
      <div className="bg-muted px-3 py-2 flex items-center justify-between">
        <span className="text-xs text-muted-foreground font-mono">{language}</span>
        <div className="flex gap-2">
          <Button
            onClick={runInlineCode}
            size="sm"
            variant="ghost"
            className="h-6 px-2 text-xs"
          >
            Run
          </Button>
          <Button
            onClick={() => setIsExpanded(!isExpanded)}
            size="sm"
            variant="ghost"
            className="h-6 px-2 text-xs"
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </Button>
        </div>
      </div>

      <div className={`transition-all duration-200 ${isExpanded ? 'max-h-96' : 'max-h-24'} overflow-auto`}>
        <pre className="p-3 text-xs">
          <code>{code}</code>
        </pre>
      </div>

      {output && (
        <div className="border-t border-border bg-background p-3">
          <pre className="text-xs text-muted-foreground">{output}</pre>
        </div>
      )}
    </div>
  );
}