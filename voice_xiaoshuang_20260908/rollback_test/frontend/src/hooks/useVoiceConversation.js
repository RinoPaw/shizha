import useBrowserDuplexVoice from "./useBrowserDuplexVoice.js";
import { REALTIME_VOICE_STATUS } from "./voiceState.js";

export { REALTIME_VOICE_STATUS };

export function useVoiceConversation(options) {
  return useBrowserDuplexVoice(options);
}

export default useVoiceConversation;
