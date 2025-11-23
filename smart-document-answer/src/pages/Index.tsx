import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FileText, Link as LinkIcon, Sparkles, Upload, Moon, Sun, Volume2, Loader2, Pause, Play } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { useTheme } from "next-themes";

type SourceType = "pdf" | "url";

const Index = () => {
  const { toast } = useToast();
  const { theme, setTheme } = useTheme();
  const [llmModel, setLlmModel] = useState<string>("mistral");
  const [embeddingModel, setEmbeddingModel] = useState<string>("all-MiniLM-L6-v2");
  const [sourceType, setSourceType] = useState<SourceType>("pdf");
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [websiteUrl, setWebsiteUrl] = useState<string>("");
  const [question, setQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [availableLlmModels, setAvailableLlmModels] = useState<string[]>(["mistral", "llava"]);
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<string[]>(["all-MiniLM-L6-v2", "all-mpnet-base-v2", "nomic-embed-text"]);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  // Fetch available models from backend on component mount
  useEffect(() => {
    const fetchAvailableModels = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (response.ok) {
          const data = await response.json();
          setAvailableLlmModels(data.llm_models || ["mistral", "llava"]);
          setAvailableEmbeddingModels(data.embedding_models || ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "nomic-embed-text"]);
          
          // Set default values if current values are not in available models
          if (!data.llm_models.includes(llmModel) && data.llm_models.length > 0) {
            setLlmModel(data.llm_models[0]);
          }
          if (!data.embedding_models.includes(embeddingModel) && data.embedding_models.length > 0) {
            setEmbeddingModel(data.embedding_models[0]);
          }
        }
      } catch (error) {
        console.error("Failed to fetch available models:", error);
        // Keep default values on error
      }
    };
    
    fetchAvailableModels();
  }, []); // Empty dependency array - only run on mount

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setPdfFile(e.target.files[0]);
      setSessionId(null); // Reset session when new file is selected
    }
  };

  const handleSourceTypeChange = (value: SourceType) => {
    setSourceType(value);
    setSessionId(null); // Reset session when source type changes
    setPdfFile(null);
    setWebsiteUrl("");
    setAnswer("");
  };

  const handleFileUpload = async () => {
    if (!pdfFile) {
      toast({
        title: "File required",
        description: "Please select a PDF file first.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", pdfFile);
      formData.append("llm_model", llmModel);
      formData.append("embedding_model", embeddingModel);

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to upload file");
      }

      const data = await response.json();
      setSessionId(data.session_id);
      
      toast({
        title: "File uploaded",
        description: "File processed successfully. You can now ask questions.",
      });
    } catch (error: any) {
      toast({
        title: "Upload failed",
        description: error.message || "Failed to upload file",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleUrlProcess = async () => {
    if (!websiteUrl.trim()) {
      toast({
        title: "URL required",
        description: "Please enter a website URL first.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/process-url`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: websiteUrl,
          llm_model: llmModel,
          embedding_model: embeddingModel,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to process URL");
      }

      const data = await response.json();
      setSessionId(data.session_id);
      
      toast({
        title: "URL processed",
        description: "Website content processed successfully. You can now ask questions.",
      });
    } catch (error: any) {
      toast({
        title: "Processing failed",
        description: error.message || "Failed to process URL",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) {
      toast({
        title: "Question required",
        description: "Please enter a question to ask.",
        variant: "destructive",
      });
      return;
    }

    if (!sessionId) {
      toast({
        title: "Document required",
        description: sourceType === "pdf" 
          ? "Please upload a PDF file first." 
          : "Please process a website URL first.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setAnswer("");
    
    try {
      const response = await fetch(`${API_BASE_URL}/query/${sessionId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
          llm_model: llmModel,
          embedding_model: embeddingModel,
          tone: "neutral",
          top_k: 5,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to generate answer");
      }

      const data = await response.json();
      setAnswer(data.answer || "");
      
      toast({
        title: "Answer generated",
        description: "Your question has been answered successfully.",
      });
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to generate answer",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleTextToSpeech = async () => {
    if (!answer.trim()) {
      toast({
        title: "No answer to read",
        description: "Please generate an answer first.",
        variant: "destructive",
      });
      return;
    }

    // If audio already exists and is paused, resume playing
    if (audioRef.current && audioUrl) {
      audioRef.current.play();
      setIsPlaying(true);
      return;
    }

    setIsGeneratingAudio(true);
    try {
      const response = await fetch(`${API_BASE_URL}/tts`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: answer,
          lang: "en",
          slow: false,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to generate audio");
      }

      // Create a blob URL from the audio response
      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);

      // Create audio element and set up event listeners
      const audio = new Audio(url);
      audioRef.current = audio;

      audio.onplay = () => setIsPlaying(true);
      audio.onpause = () => setIsPlaying(false);
      audio.onended = () => {
        setIsPlaying(false);
        // Clean up
        URL.revokeObjectURL(url);
        setAudioUrl(null);
        audioRef.current = null;
      };

      // Auto-play the audio
      audio.play().catch((error) => {
        console.error("Error playing audio:", error);
        toast({
          title: "Playback error",
          description: "Could not play audio. Please try again.",
          variant: "destructive",
        });
        setIsGeneratingAudio(false);
      });

      toast({
        title: "Audio generated",
        description: "Playing the answer audio.",
      });
    } catch (error: any) {
      toast({
        title: "Audio generation failed",
        description: error.message || "Failed to generate audio",
        variant: "destructive",
      });
    } finally {
      setIsGeneratingAudio(false);
    }
  };

  const handlePausePlay = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <div className="min-h-screen font-body" style={{ background: 'var(--gradient-bg)' }}>
      <div className="container mx-auto px-4 py-8 lg:py-12">
        {/* Header */}
        <div className="mb-8 text-center animate-fade-in">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Sparkles className="w-8 h-8 text-primary" />
            <h1 className="text-4xl lg:text-5xl font-heading font-bold bg-gradient-primary bg-clip-text text-transparent">
              DocuMind AI
            </h1>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="ml-4"
            >
              <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Ask questions about your documents using advanced AI models
          </p>
        </div>

        <div className="space-y-6 max-w-7xl mx-auto">
          {/* Configuration Bar */}
          <Card className="p-6 shadow-soft backdrop-blur-sm bg-card/95 animate-slide-in">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* LLM Model Selection */}
              <div className="space-y-2">
                <Label htmlFor="llm-model" className="text-sm font-medium">
                  LLM Model
                </Label>
                <Select 
                  value={llmModel} 
                  onValueChange={(value) => {
                    setLlmModel(value);
                    setSessionId(null); // Reset session when model changes
                  }}
                >
                  <SelectTrigger id="llm-model" className="bg-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover z-50">
                    {availableLlmModels.map((model) => (
                      <SelectItem key={model} value={model}>
                        {model.charAt(0).toUpperCase() + model.slice(1)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Embedding Model Selection */}
              <div className="space-y-2">
                <Label htmlFor="embedding-model" className="text-sm font-medium">
                  Embedding Model
                </Label>
                <Select 
                  value={embeddingModel} 
                  onValueChange={(value) => {
                    setEmbeddingModel(value);
                    setSessionId(null); // Reset session when embedding model changes
                  }}
                >
                  <SelectTrigger id="embedding-model" className="bg-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover z-50">
                    {availableEmbeddingModels.map((model) => (
                      <SelectItem key={model} value={model}>
                        {model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Source Type Selection */}
              <div className="space-y-2">
                <Label htmlFor="source-type" className="text-sm font-medium">
                  Source Type
                </Label>
                <Select
                  value={sourceType}
                  onValueChange={(value) => handleSourceTypeChange(value as SourceType)}
                >
                  <SelectTrigger id="source-type" className="bg-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover z-50">
                    <SelectItem value="pdf">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4" />
                        PDF Upload
                      </div>
                    </SelectItem>
                    <SelectItem value="url">
                      <div className="flex items-center gap-2">
                        <LinkIcon className="w-4 h-4" />
                        Website URL
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Dynamic Source Input */}
              <div className="space-y-2">
                <Label className="text-sm font-medium">
                  {sourceType === "pdf" ? "Upload PDF" : "Website URL"}
                </Label>
                {sourceType === "pdf" ? (
                  <div className="space-y-2">
                    <div className="relative">
                      <Input
                        type="file"
                        accept=".pdf,.docx,.txt"
                        onChange={handleFileChange}
                        className="hidden"
                        id="pdf-upload"
                      />
                      <Label
                        htmlFor="pdf-upload"
                        className="flex items-center justify-center gap-2 h-10 px-3 border-2 border-dashed border-border rounded-lg cursor-pointer hover:border-primary transition-colors bg-background"
                      >
                        <Upload className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground truncate">
                          {pdfFile ? pdfFile.name : "Upload Document"}
                        </span>
                      </Label>
                    </div>
                    {pdfFile && !sessionId && (
                      <Button
                        onClick={handleFileUpload}
                        disabled={isLoading}
                        size="sm"
                        className="w-full"
                      >
                        {isLoading ? "Processing..." : "Process Document"}
                      </Button>
                    )}
                    {sessionId && (
                      <div className="text-xs text-green-600 dark:text-green-400">
                        ✓ Document ready
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Input
                      type="url"
                      placeholder="https://example.com"
                      value={websiteUrl}
                      onChange={(e) => setWebsiteUrl(e.target.value)}
                      className="bg-background"
                      disabled={isLoading}
                    />
                    {websiteUrl && !sessionId && (
                      <Button
                        onClick={handleUrlProcess}
                        disabled={isLoading}
                        size="sm"
                        className="w-full"
                      >
                        {isLoading ? "Processing..." : "Process URL"}
                      </Button>
                    )}
                    {sessionId && (
                      <div className="text-xs text-green-600 dark:text-green-400">
                        ✓ URL processed
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </Card>

          {/* Main Q&A Area */}
          <div className="space-y-6 animate-fade-in" style={{ animationDelay: "0.1s" }}>
            {/* Question Input */}
            <Card className="p-6 shadow-soft backdrop-blur-sm bg-card/95">
              <Label htmlFor="question" className="text-lg font-heading font-semibold mb-3 block">
                Ask Your Question
              </Label>
              <Textarea
                id="question"
                placeholder="What would you like to know about this document?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="min-h-[120px] mb-4 bg-background resize-none"
              />
              <Button
                onClick={handleAsk}
                disabled={isLoading}
                className="w-full bg-gradient-primary hover:shadow-glow transition-all duration-300 font-medium"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <span className="animate-pulse">Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Ask Question
                  </>
                )}
              </Button>
            </Card>

            {/* Results Display */}
            {answer && (
              <Card className="p-6 shadow-soft backdrop-blur-sm bg-card/95 animate-fade-in">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-heading font-semibold flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-gradient-accent"></span>
                    Answer
                  </h3>
                  <div className="flex items-center gap-2">
                    {audioUrl && (
                      <Button
                        onClick={handlePausePlay}
                        disabled={!audioUrl}
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-2"
                      >
                        {isPlaying ? (
                          <>
                            <Pause className="w-4 h-4" />
                            Pause
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4" />
                            Play
                          </>
                        )}
                      </Button>
                    )}
                    <Button
                      onClick={handleTextToSpeech}
                      disabled={isGeneratingAudio || !answer.trim()}
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2"
                    >
                      {isGeneratingAudio ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Volume2 className="w-4 h-4" />
                          Read Aloud
                        </>
                      )}
                    </Button>
                  </div>
                </div>

                {/* Answer */}
                <div className="prose prose-sm max-w-none">
                  <div className="bg-background/50 rounded-lg p-5 max-h-[500px] overflow-y-auto custom-scrollbar">
                    <p className="text-foreground whitespace-pre-wrap leading-relaxed">
                      {answer}
                    </p>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: hsl(var(--muted));
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: hsl(var(--primary));
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: hsl(var(--primary) / 0.8);
        }
      `}</style>
    </div>
  );
};

export default Index;
