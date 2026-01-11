import { Container, Stack, Title, Text } from "@mantine/core";
import { LiveVisualizationSection } from "../components/LiveVisualization";

export function Visualizations() {
  return (
    <Container size="xl" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Interactive Visualizations</Title>
          <Text size="lg" c="dimmed">
            Explore Amari's mathematical concepts through real-time interactive demonstrations
          </Text>
        </div>

        <LiveVisualizationSection />
      </Stack>
    </Container>
  );
}
