import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import StockSearch from './pages/StockSearch';
import StockDetail from './pages/StockDetail';
import ModelManagement from './pages/ModelManagement';
import Navigation from './components/Navigation';

const { Header, Content } = Layout;

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Layout style={{ minHeight: '100vh' }}>
        <Header>
          <Navigation />
        </Header>
        <Content style={{ padding: '24px' }}>
          <Routes>
            <Route path="/" element={<StockSearch />} />
            <Route path="/stock/:symbol" element={<StockDetail />} />
            <Route path="/models" element={<ModelManagement />} />
          </Routes>
        </Content>
      </Layout>
    </BrowserRouter>
  );
};

export default App;
