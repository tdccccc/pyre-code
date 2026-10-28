import { NextResponse } from 'next/server';

interface ModelsRequest {
  baseUrl?: string;
  apiKey?: string;
}

/* Proxies GET {baseUrl}/models so the browser avoids CORS. Only uses client-supplied
   credentials — the server env key is never sent to a user-provided URL. */
export async function POST(request: Request) {
  const body = (await request.json().catch(() => ({}))) as ModelsRequest;
  const baseUrl = body.baseUrl?.trim().replace(/\/+$/, '') || '';
  const apiKey = body.apiKey?.trim() || '';

  if (!baseUrl || !apiKey) {
    return NextResponse.json({ error: 'Missing baseUrl or apiKey.' }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${baseUrl}/models`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    const data = await upstream.json().catch(() => null);
    if (!upstream.ok) {
      return NextResponse.json(
        { error: `Upstream request failed with status ${upstream.status}.` },
        { status: upstream.status === 401 ? 401 : 502 },
      );
    }

    const list: unknown[] = Array.isArray(data?.data) ? data.data : Array.isArray(data) ? data : [];
    const models = list
      .map((item) => (typeof item === 'string' ? item : (item as { id?: unknown })?.id))
      .filter((id): id is string => typeof id === 'string' && id.length > 0);

    return NextResponse.json({ models: [...new Set(models)].sort() });
  } catch {
    return NextResponse.json({ error: 'Failed to reach the configured AI endpoint.' }, { status: 502 });
  }
}
