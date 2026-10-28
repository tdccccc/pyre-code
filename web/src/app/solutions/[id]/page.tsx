import { TopNav } from '@/components/layout/TopNav';
import { Footer } from '@/components/layout/Footer';
import { SolutionPageContent } from '@/components/solution/SolutionPage';
import problems from '@/lib/problems.json';

export default async function SolutionsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  // Server-side lookup, so only the track reaches the client: Hot 100 breadcrumbs lead to /hot100
  const problem = problems.problems.find((p) => p.id === id);
  const isHot100 = !!problem && 'track' in problem && problem.track === 'hot100';

  return (
    <div className="min-h-screen bg-bg">
      <TopNav />
      <main className="max-w-[1280px] mx-auto px-7 pt-8 pb-20">
        <SolutionPageContent problemId={id} listHref={isHot100 ? '/hot100' : '/problems'} />
      </main>
      <Footer />
    </div>
  );
}
