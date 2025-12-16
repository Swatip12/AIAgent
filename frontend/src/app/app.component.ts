import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';

type ChatRole = 'assistant' | 'user';

interface ChatMessage {
  role: ChatRole;
  text: string;
}

interface LessonStepResponse {
  session_id: string;
  step: string;
  checkpoint_question: string;
  recap: string;
}

interface PracticeItem {
  question: string;
  kind: string;
}

interface PracticeResponse {
  session_id: string;
  practice: PracticeItem[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'AI Teaching Assistant';
  apiBase = 'http://localhost:8000';

  subjects = ['Java', 'Logical Reasoning', 'Aptitude', 'Data Structures', 'Full Stack Development'];
  levels = ['beginner', 'intermediate'];

  selectedSubject = this.subjects[0];
  topic = 'Classes and Objects';
  level = this.levels[0];
  sessionId?: string;

  messages: ChatMessage[] = [];
  practice: PracticeItem[] = [];

  lastAnswer = '';
  isLoading = false;
  status = '';

  constructor(private http: HttpClient) {}

  startLesson() {
    this.messages = [];
    this.practice = [];
    this.sessionId = undefined;
    this.sendLessonStep();
  }

  sendLessonStep(confusion = false) {
    this.isLoading = true;
    this.status = 'Guiding...';
    const payload = {
      subject: this.selectedSubject,
      topic: this.topic,
      level: this.level,
      session_id: this.sessionId,
      last_answer: this.lastAnswer || undefined,
      confusion,
    };

    this.http.post<LessonStepResponse>(`${this.apiBase}/lesson-step`, payload).subscribe({
      next: (res) => {
        this.sessionId = res.session_id;
        const combined = `${res.step}\n\n${res.checkpoint_question}\n${res.recap}`;
        this.messages = [...this.messages, { role: 'assistant', text: combined }];
        this.lastAnswer = '';
        this.isLoading = false;
        this.status = '';
      },
      error: (err) => {
        this.isLoading = false;
        this.status = err?.error?.detail || 'Something went wrong';
      },
    });
  }

  markConfused() {
    this.sendLessonStep(true);
  }

  generatePractice() {
    if (!this.sessionId) {
      this.status = 'Start a lesson first.';
      return;
    }
    this.isLoading = true;
    this.status = 'Building practice...';
    const payload = {
      subject: this.selectedSubject,
      topic: this.topic,
      level: this.level,
      session_id: this.sessionId,
    };

    this.http.post<PracticeResponse>(`${this.apiBase}/practice`, payload).subscribe({
      next: (res) => {
        this.practice = res.practice;
        this.sessionId = res.session_id;
        this.isLoading = false;
        this.status = '';
      },
      error: (err) => {
        this.isLoading = false;
        this.status = err?.error?.detail || 'Could not fetch practice';
      },
    });
  }
}
