import { useState, useCallback } from "react";
import { Card, CardHeader, CardBody, Button, CodeBlock, TextArea, H3, P } from "jadis-ui";
import { ErrorBoundary } from "./ErrorBoundary";
import { safeExecute, validateNumbers, validateArray } from "../utils/safeExecution";

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

    const executionResult = await safeExecute(
      async () => {
        const result = await onRun(code);
        return result;
      },
      {
        timeout: 15000,
        retries: 1,
        fallback: () => ({
          output: "Code execution timed out or failed. This might be due to infinite loops, heavy computations, or syntax errors.",
          error: "Execution timeout"
        })
      }
    );

    setIsRunning(false);

    if (executionResult.success && executionResult.data) {
      setOutput(executionResult.data.output);
      if (executionResult.data.error) {
        setError(executionResult.data.error);
      }
      if (executionResult.fallbackUsed) {
        setError("Execution failed - fallback message shown");
      }
    } else {
      setError(executionResult.error || "Unknown execution error");
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
    <ErrorBoundary>
      <Card>
        <CardHeader>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <H3>{title}</H3>
            {description && (
              <P style={{ fontSize: '0.875rem', marginTop: '0.25rem', opacity: 0.7 }}>{description}</P>
            )}
          </div>
          {onRun && (
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <Button
                onClick={runCode}
                disabled={isRunning}
                title="Run code (Ctrl/Cmd + Enter)"
              >
                {isRunning ? 'Running...' : 'Run'}
              </Button>
              <Button
                onClick={() => setCode(initialCode)}
                variant="outline"
                title="Reset to initial code"
              >
                Reset
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardBody>
        <div>
          <TextArea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            onKeyDown={handleKeyDown}
            style={{
              fontFamily: 'monospace',
              fontSize: '0.875rem',
              minHeight: height === 'h-64' ? '16rem' : '10rem'
            }}
            spellCheck={false}
            placeholder="Enter your code here..."
          />
        </div>

        {/* Output Section */}
        {(output || error) && (
          <div style={{ marginTop: '1rem' }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>Output:</h4>
            <CodeBlock
              language="text"
              variant={error ? 'error' : 'success'}
            >
              {error || output}
            </CodeBlock>
          </div>
        )}
      </CardBody>
    </Card>
    </ErrorBoundary>
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
    <Card style={{ marginTop: '0.75rem', marginBottom: '0.75rem' }}>
      <CardHeader style={{ padding: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', opacity: 0.7 }}>{language}</span>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <Button
              onClick={runInlineCode}
              variant="ghost"
              style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            >
              Run
            </Button>
            <Button
              onClick={() => setIsExpanded(!isExpanded)}
              variant="ghost"
              style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            >
              {isExpanded ? 'Collapse' : 'Expand'}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardBody style={{ padding: '0' }}>
        <div style={{
          transition: 'all 0.2s',
          maxHeight: isExpanded ? '24rem' : '6rem',
          overflow: 'auto'
        }}>
          <CodeBlock
            language={language}
            showLineNumbers={true}
            showCopyButton={true}
          >
            {code}
          </CodeBlock>
        </div>

        {output && (
          <div style={{ borderTop: '1px solid var(--border)', padding: '0.75rem' }}>
            <CodeBlock language="text" variant="muted">
              {output}
            </CodeBlock>
          </div>
        )}
      </CardBody>
    </Card>
  );
}