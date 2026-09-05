import { NextResponse } from "next/server"

export function localDigitizerUiEnabled() {
  return process.env.ECG_DIGITIZER_ENABLE_LOCAL_UI === "true"
}

export function rejectNonLocalDigitizerRequest(request: Request) {
  const requestUrl = new URL(request.url)
  const isLoopback = isLoopbackHostname(requestUrl.hostname)
  const origin = request.headers.get("origin")
  const sameOrigin = isSameLocalOrigin(requestUrl, origin)
  const mutationHeaderAccepted =
    request.method === "GET" ||
    request.method === "HEAD" ||
    request.headers.get("x-ecg-digitizer-local") === "1"

  if (localDigitizerUiEnabled() && isLoopback && sameOrigin && mutationHeaderAccepted) {
    return null
  }

  return NextResponse.json({ error: "Not found." }, { status: 404 })
}

function isSameLocalOrigin(requestUrl: URL, origin: string | null) {
  if (!origin || origin === requestUrl.origin) return true

  try {
    const originUrl = new URL(origin)
    return (
      isLoopbackHostname(requestUrl.hostname) &&
      isLoopbackHostname(originUrl.hostname) &&
      originUrl.protocol === requestUrl.protocol &&
      originUrl.port === requestUrl.port
    )
  } catch {
    return false
  }
}

function isLoopbackHostname(hostname: string) {
  return (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname === "[::1]"
  )
}
