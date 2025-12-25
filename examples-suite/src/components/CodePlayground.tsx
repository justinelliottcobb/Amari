import { useState, useCallback } from "react";
import { Card, Title, Text, Button, Textarea, Group, Stack, Box } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import { ErrorBoundary } from "./ErrorBoundary";
import { safeExecute } from "../utils/safeExecution";

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

  return (
    <ErrorBoundary>
      <Card withBorder>
        <Card.Section withBorder inheritPadding py="sm">
          <Group justify="space-between" align="flex-start">
            <Box>
              <Title order={4}>{title}</Title>
              {description && (
                <Text size="sm" c="dimmed" mt={4}>{description}</Text>
              )}
            </Box>
            {onRun && (
              <Group gap="xs">
                <Button
                  onClick={runCode}
                  disabled={isRunning}
                  loading={isRunning}
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
              </Group>
            )}
          </Group>
        </Card.Section>
        <Card.Section inheritPadding py="md">
          <Stack gap="md">
            <Textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              onKeyDown={handleKeyDown}
              styles={{
                input: {
                  fontFamily: 'var(--mantine-font-family-monospace)',
                  fontSize: '0.875rem',
                  minHeight: height === 'h-64' ? '16rem' : '10rem'
                }
              }}
              spellCheck={false}
              placeholder="Enter your code here..."
              autosize
              minRows={8}
            />

            {/* Output Section */}
            {(output || error) && (
              <Box>
                <Text size="sm" fw={600} mb="xs">Output:</Text>
                <Box
                  p="sm"
                  style={{
                    backgroundColor: error
                      ? 'rgba(239, 68, 68, 0.1)'
                      : 'rgba(34, 197, 94, 0.1)',
                    borderRadius: 'var(--mantine-radius-sm)',
                    border: `1px solid ${error ? 'var(--mantine-color-red-6)' : 'var(--mantine-color-green-6)'}`
                  }}
                >
                  <Text
                    size="sm"
                    ff="monospace"
                    c={error ? 'red' : 'green'}
                    style={{ whiteSpace: 'pre-wrap' }}
                  >
                    {error || output}
                  </Text>
                </Box>
              </Box>
            )}
          </Stack>
        </Card.Section>
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
    <Card withBorder my="sm">
      <Card.Section withBorder inheritPadding py="xs">
        <Group justify="space-between">
          <Text size="xs" ff="monospace" c="dimmed">{language}</Text>
          <Group gap="xs">
            <Button
              onClick={runInlineCode}
              variant="subtle"
              size="xs"
            >
              Run
            </Button>
            <Button
              onClick={() => setIsExpanded(!isExpanded)}
              variant="subtle"
              size="xs"
            >
              {isExpanded ? 'Collapse' : 'Expand'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section p={0}>
        <Box
          style={{
            maxHeight: isExpanded ? '24rem' : '6rem',
            overflow: 'auto',
            transition: 'max-height 0.2s ease'
          }}
        >
          <CodeHighlight
            code={code}
            language={language}
            withCopyButton
          />
        </Box>

        {output && (
          <Box
            p="sm"
            style={{ borderTop: '1px solid var(--mantine-color-dark-4)' }}
          >
            <Text size="xs" ff="monospace" c="dimmed" style={{ whiteSpace: 'pre-wrap' }}>
              {output}
            </Text>
          </Box>
        )}
      </Card.Section>
    </Card>
  );
}
