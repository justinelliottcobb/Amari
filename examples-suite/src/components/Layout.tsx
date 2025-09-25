import { PageLayout, Sidebar, Grid, GridItem } from "jadis-ui";
import { Navigation, MobileNavigation } from "./Navigation";

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <PageLayout>
      <Grid columns={2} gap="none">
        <GridItem span={1}>
          <Sidebar>
            <Navigation />
          </Sidebar>
        </GridItem>
        <GridItem span={1}>
          <main>
            {children}
          </main>
        </GridItem>
      </Grid>
    </PageLayout>
  );
}