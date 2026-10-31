import { cookies } from 'next/headers';

/**
 * The learner's session token. Normally an anonymous per-browser cookie; setting
 * PYRE_SESSION_TOKEN pins every browser to one learner, for a single-person
 * deployment opened from several devices.
 */
export async function getSessionToken(): Promise<string | undefined> {
  const pinned = process.env.PYRE_SESSION_TOKEN;
  if (pinned) return pinned;
  const cookieStore = await cookies();
  return cookieStore.get('session_token')?.value;
}
